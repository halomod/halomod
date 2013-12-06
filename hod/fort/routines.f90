!! FORTRAN ROUTINES TO DO QUICK WORK ON LOOPS FOR HOD
module hod_routines
contains
subroutine simps(dx, func,val)
    implicit none
    !!simps works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out) :: val !The value of the integral

    integer :: n !length of func
    integer :: i,start
    real(8) :: endbit

    n = size(func)
    if (mod(n,2)>0) then
        start = 2
        endbit = (func(1)+func(2))*dx/2
    else
        start = 1
        endbit=0.d0
    end if

    val = func(start) + func(n) + 4*func(n-1)
    do i=start,n/2-1
        val = val + 2*func(2*i)
        val = val + 4*func(2*i-1)
    end do
    val = val*dx/3 + endbit
end subroutine

subroutine trapz(dx,func,val)
    implicit none
    !!trapz works on functions defined at EQUIDISTANT values of x only

    real(8), intent(in) :: dx !The grid spacing
    real(8), intent(in) :: func(:) !The function
    real(8), intent(out) :: val !The value of the integral

    integer :: n !length of func
    integer :: i

    val = func(1)+func(n)
    do i=2,n-1
        val = val + 2*func(i)
    end do
    val = val*dx/2
end subroutine

!!!!!!!!!!!!! START ACTUAL CALLED ROUTINES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine power_gal_2h(nlnk,nm,u,bias,nsat,ncen,dndm,mass,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: bias(nm), nsat(nm),ncen(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)

    real(8), intent(out):: power(nlnk)

    real(8) ::mass_part(nm)
    integer :: i
    real(8) :: integrand(nm),dm

    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    mass_part = ncen*dndm*bias*mass
    do i=1,nlnk
        integrand = (1.d0+u(i,:) *nsat)*mass_part !multiply by mass as we integrate in log-space
        call simps(dm,integrand,power(i))
      !  if (i==nlnk) then
      !      write(*,*) maxval(integrand),minval(integrand)
      !      write(*,*) maxval(mass_part/mass),minval(mass_part/mass)
      !      write(*,*) maxval(u(i,:)),minval(u(i,:))
      !  end if
    end do

end subroutine

subroutine power_gal_1h_ss(nlnk,nm,u,dndm,nsat,ncen,mass,central,power)
    implicit none

    integer, intent(in) :: nlnk, nm !Size of lnk and mass arrays respectively
    real(8), intent(in) :: ncen(nm),nsat(nm),dndm(nm), mass(nm)
    real(8), intent(in) :: u(nlnk,nm)
    logical, intent(in) :: central

    real(8), intent(out):: power(nlnk)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0

    dm = log(mass(2))-log(mass(1))

    do i=1,nlnk
        if (central)then
            integrand = ncen*nsat**2 * u(i,:)**2*dndm*mass
        else
            integrand = nsat**2 * u(i,:)**2 *dndm*mass
        end if
        call simps(dm,integrand,power(i))
    end do
end subroutine

subroutine corr_gal_1h_ss(nr,nm,mass,dndm,ncen,nsat,lam,central,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: integrand(nm),dm
    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)/mass !Not mass**2 since int in log space
        else
            integrand = dndm*nsat**2*lam(i,:)/mass
        end if

        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h_cs(nr,nm,r,mass,dndm,ncen,nsat,rho,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: mmin
    real(8) :: integrand(nm),dm
    integrand = 0.d0

    dm = log(mass(2))-log(mass(1))

    do i=1,nr
        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3
        where (mass>mmin) integrand = dndm*2*ncen*nsat*rho(i,:)*mass !*mass since in logspace
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine corr_gal_1h(nr,nm,r,mass,dndm,ncen,nsat,rho,lam,central,mean_dens,delta_halo,corr)
    implicit none

    integer, intent(in) :: nr,nm
    real(8), intent(in) :: r(nr), mass(nm), ncen(nm),nsat(nm),dndm(nm)
    real(8), intent(in) :: rho(nr,nm),mean_dens,delta_halo
    real(8), intent(in) :: lam(nr,nm)
    logical, intent(in) :: central

    real(8), intent(out) :: corr(nr)

    integer :: i,j
    real(8) :: mmin
    real(8) :: integrand(nm),dm

    integrand = 0.d0
    dm = log(mass(2))-log(mass(1))

    !Integrating in log-space, so everything is multiplied by mass
    do i=1,nr
        if (central)then
            integrand = dndm*ncen*nsat**2*lam(i,:)/mass
        else
            integrand = dndm*nsat**2*lam(i,:)/mass
        end if

        mmin = 4*3.14159265*r(i)**3*mean_dens*delta_halo/3
        where (mass>mmin) integrand = integrand+dndm*2*ncen*nsat*rho(i,:)*mass
     !   if (i==1)then
     !       do j=1,nm
                !if (mass(j)/0.7>1e13 .and. mass(j)/0.7<1.2e13)then
                !    write(*,*) mass(j)/1e13, integrand(j),nsat(j),ncen(j),dndm(j),rho(1,j),lam(1,j)
                    !"(I3,8ES20.6)"
                !end if
      !      end do
      !  end if
        call simps(dm,integrand,corr(i))
    end do
end subroutine

subroutine power_to_corr(nlnk,nr,lnk,r,power,corr)
    implicit none
    !This routine is a bit of a hack really. We stop integrating when
    !we can no longer fit 5 dlnk within an oscillation.
    !There is no saying that the value we end up on should be the correct one.
    !If maxk is large, then we should be fine, but if not its a bit luck-of-the-draw.
    integer, intent(in) :: nlnk, nr
    real(8), intent(in) :: r(nr),power(nlnk),lnk(nlnk)

    real(8), intent(out) :: corr(nr)

    integer :: i
    real(8) :: k(nlnk), integrand(nlnk),dlnk,maxk
    integrand = 0.d0

    k = exp(lnk)
    dlnk = lnk(2) - lnk(1)

    do i=1,nr
        !maxk = 3.14159265/(5*r(i)*(exp(lnk(2)-lnk(1))-1))
        !where (k < maxk)
        integrand = power*k**2*sin(k*r(i))/r(i)
        call simps(dlnk,integrand,corr(i))
    end do

    corr = corr/(2*3.141592653589**2)
end subroutine
end module
